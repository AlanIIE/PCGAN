from conditional_gan import make_generator, make_discriminator, CGAN, generate_pose
import cmd
from gan.train import Trainer

from pose_dataset import PoseHMDataset
# from keras.utils import multi_gpu_model

def main():
    args = cmd.args()

    generator = make_generator(args.image_size, args.use_input_pose, args.warp_agg, args.num_landmarks, args.num_mask)
    generator.summary()
    if args.generator_checkpoint is not None:
        generator.load_weights(args.generator_checkpoint, by_name=True)

    discriminator = make_discriminator(args.image_size, args.use_input_pose, args.num_landmarks, args.num_mask)
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)

    dataset = PoseHMDataset(test_phase=False, **vars(args))
    gan = CGAN(generator, discriminator, **vars(args))

    trainer = Trainer(dataset, gan, **vars(args))
    
    trainer.train()
    
if __name__ == "__main__":
    main()
