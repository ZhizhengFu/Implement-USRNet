import os
import logging
import random
import math
import lpips
import wandb
from datetime import datetime
from torch.utils.data import DataLoader

from utils.utils_dataset import define_Dataset
from models.usrnet_train import define_Model
import utils.utils_option as option
import utils.utils_image as util
import utils.utils_logger as utils_logger
from utils.utils_training import seed_everywhere, save_code_snapshot


def main():
    wandbconfig = False
    opt = option.parse('./options/train_usrnet.json')
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    optim_name = f'bs{opt["datasets"]["train"]["dataloader_batch_size"]}-loss_{opt["train"]["G_lossfn_type"]}-lr_{opt["train"]["G_optimizer_lr"]}-G_scheduler_milestones_{opt["train"]["G_scheduler_milestones"]}'
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    expr_name = f'USRNet-{run_id}-{optim_name}'
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    # opt['path']['pretrained_netG'] = init_path_G
    current_step = 0
    border = opt['scale']

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    if wandbconfig:
        wandb.init(
            project="USRNet",
            name=expr_name,
            config=opt,
        )

    # ----------------------------------------
    # save code snapshot
    # ----------------------------------------
    save_code_snapshot(os.path.join(opt['path']['log'], 'codes'))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    seed_everywhere(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    lpips_model = lpips.LPIPS(net='vgg').eval().to(model.device)

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)
            logs = model.current_log()
            if wandbconfig:
                wandb.log(logs)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                if wandbconfig:
                    wandb.log({"epoch": epoch, "iter": current_step, "lr": model.current_learning_rate()})
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                avg_psnr = 0.0
                avg_lpips = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    # -----------------------
                    # calculate LPIPS
                    # -----------------------
                    lpips_value = lpips_model.forward(visuals['E'].to(model.device), visuals['H'].to(model.device)).item()
                    print(lpips_value)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB | LPIPS: {:.3f}'.format(idx, image_name_ext, current_psnr, lpips_value))

                    avg_psnr += current_psnr
                    avg_lpips += lpips_value

                avg_psnr = avg_psnr / idx
                avg_lpips = avg_lpips / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average LPIPS: {:.3f}\n'.format(epoch, current_step, avg_psnr, avg_lpips))
                if wandbconfig:
                    wandb.log({"iter": current_step, "ave_PSNR": avg_psnr, "ave_LPIPS": avg_lpips})

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    if wandbconfig:
        wandb.finish()


if __name__ == '__main__':
    main()
