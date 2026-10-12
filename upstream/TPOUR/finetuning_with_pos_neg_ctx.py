import os
import torch
import logging
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.options import Options
from src import slurm, dist_utils, utils, contriever, finetuning_data, inbatch, moco
import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def finetuning(opt, model, optimizer, scheduler, tokenizer, step):
    trained_step = step
    step = 0
    run_stats = utils.WeightedAvgStats()
    tb_logger = utils.init_tb_logger(opt.output_dir)

    print("Loading training data (Est 3 min - 1,500,000it)")
    train_dataset = finetuning_data.PositiveDataset(
        datapaths=opt.train_data,
        negative_ctxs=opt.negative_ctxs,
        negative_hard_ratio=opt.negative_hard_ratio,
        negative_hard_min_idx=opt.negative_hard_min_idx,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        timestamp_injection=opt.timestamp_injection,
        training=True,
        tokenizer=tokenizer,
        opt=opt,
    )

    if len(opt.eval_data) != 0:
        dev_dataset = finetuning_data.PositiveDataset(
            datapaths=opt.eval_data,
            negative_ctxs=opt.negative_ctxs,
            negative_hard_ratio=opt.negative_hard_ratio,
            negative_hard_min_idx=opt.negative_hard_min_idx,
            normalize=opt.eval_normalize_text,
            global_rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            timestamp_injection=opt.timestamp_injection,
            maxload=50000,
            training=False,
            tokenizer=tokenizer,
            opt=opt,
        )

    collator = finetuning_data.PositiveCollator(tokenizer, chunk_length=opt.chunk_length, opt=opt, passage_maxlength=opt.chunk_length)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        collate_fn=collator,
    )
    if len(opt.eval_data) != 0:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            batch_size=opt.per_gpu_batch_size,
            drop_last=False,
            collate_fn=collator,
        )

    epoch = 1
    model.train()
    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        for i, batch in enumerate(train_dataloader):
            step += 1
            if step <= trained_step:
                print(f"skip step {step} until {trained_step}")
                continue

            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            train_loss, iter_stats = model(**batch, stats_prefix="train", scheduler=scheduler)
            train_loss.backward(retain_graph=True)

            if opt.optim == "sam" or opt.optim == "asam":
                optimizer.first_step(zero_grad=True)

                sam_loss, _ = model(**batch, stats_prefix="train/sam_opt", scheduler=scheduler)
                sam_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    if "train" in k:
                        log += f" | {k}: {v:.3f}"
                        if tb_logger:
                            tb_logger.add_scalar(k, v, step)
                tb_logger.add_scalar("learning rate", scheduler.get_last_lr()[0], step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()
            
            if opt.eval_freq != 0 and step % opt.eval_freq == 0:
                gc.collect()
                encoder = model.get_encoder()
                if step % opt.save_freq == 0 and dist_utils.get_rank() == 0:
                    utils.save(
                        model,
                        optimizer,
                        scheduler,
                        step,
                        opt,
                        opt.output_dir,
                        f"step-{step}",
                    )
                model.train()

        epoch += 1

def main():
    # need to implement utils.load()
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()

    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    if opt.contrastive_mode == "tpour":
        model_class = moco.TPOUR
    else:
        raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")
    if not directory_exists and opt.model_path == "none":
        model = model_class(opt)
        model = model.cuda()
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
    elif not directory_exists and opt.model_path != "none":
        model, _, _, _, _ = utils.load(
            model_class,
            opt.model_path,
            opt,
            reset_params=False,
        )
        logger.info(f"2nd Phase Model loaded from {opt.output_dir}")
        step = 0
        optimizer, scheduler = utils.set_optim(opt, model)
    elif directory_exists:
        model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            model_path,
            opt,
            reset_params=False,
        )
        logger.info(f"Model loaded from {opt.output_dir}")
    else:
        retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
        opt.retriever_model_id = retriever_model_id
 
        if opt.contrastive_mode == 'inbatch':
            model = inbatch.InBatch(opt, retriever, tokenizer)
        else:
            raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")
        model = model.cuda()
        optimizer, scheduler = utils.set_optim(opt, model)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = opt.dropout

    logger.info(utils.get_parameters(model))

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
        dist.barrier()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)


if __name__ == "__main__":
    main()
