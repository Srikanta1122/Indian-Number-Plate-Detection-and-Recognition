import os
import tensorflow as tf

# Set environment variable to suppress the message
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check available physical GPUs
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    # Assuming you want to use the first GPU
    try:
        # Set virtual device configuration
        tf.config.experimental.set_virtual_device_configuration(physical_gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Adjust memory limit as needed
    except RuntimeError as e:
        print(e)
else:
    print("No physical GPUs available, cannot set virtual device configuration.")
