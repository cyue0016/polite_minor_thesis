import tensorflow as tf

# Inspecting the model checkpoint
def inspect_checkpoint(ckpt_path):
    with tf.Graph().as_default():
        # Load the meta graph and clear the device assignments
        saver = tf.train.import_meta_graph(ckpt_path + ".meta", clear_devices=True)

        with tf.Session() as sess:
            # Restore the checkpoint
            saver.restore(sess, ckpt_path)
            print("Model restored from checkpoint.")

            # Iterate through and print out all the tensors in the checkpoint
            for tensor in tf.get_default_graph().get_operations():
                print(tensor.name)

if __name__ == "__main__":
    # Replace with the path to your checkpoint directory
    ckpt_path = 'ckpt/politeness_classifier_2'
    inspect_checkpoint(ckpt_path)
