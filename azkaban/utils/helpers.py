import tensorflow as tf


class NoOpContextManager(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class TBSummaryWriter(object):
    """Logging in tensorboard without tensorflow ops"""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """
        Log a scalar variable.
        :param tag: str Name of the scalar
        :param value: int Scalar value
        :param step: int Training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def close(self):
        self.writer.close()
