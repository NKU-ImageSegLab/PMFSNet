import os.path

from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, opt):
        self.opt = opt
        try:
            base_path = opt["tensorboard_dir"]

            self.writer = SummaryWriter(os.path.join(base_path, opt["dataset_name"], opt["model_name"]).__str__())
        except:
            self.writer = None
            pass

    def write(self, summary):
        raise NotImplementedError('write method is not implemented')

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_scalar(
            self,
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
    ):
        if self.writer is not None:
            self.writer.add_scalar(
                tag, scalar_value, global_step, walltime, new_style, double_precision
            )

    def close(self):
        if self.writer is not None:
            self.writer.close()
