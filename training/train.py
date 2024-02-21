import sys
from dataloader.get_motion_data import get_motion_data


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "/home/aaryaman/Developer/Truebone_Z-OO/Ant/__Attack.bvh"
    motion_data = get_motion_data(filename)
    print(motion_data.shape)
    motion_data.to('cuda')


if __name__ == "__main__":
    main()
