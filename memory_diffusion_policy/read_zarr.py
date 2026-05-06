import zarr
import numpy as np
import sys

def explore_zarr(path):
    root = zarr.open(path, mode='r')
    
    def print_tree(group, indent=0):
        prefix = "  " * indent
        
        # Print group attributes
        if group.attrs:
            print(f"{prefix}[attrs]: {dict(group.attrs)}")
        
        for name, item in group.items():
            if isinstance(item, zarr.Group):
                print(f"{prefix}📁 {name}/")
                print_tree(item, indent + 1)
            else:  # Array
                print(f"{prefix}📄 {name}: shape={item.shape}, dtype={item.dtype}")

                if name in ['state']:
                    # Show sample data
                    if item.size > 0:
                        print(f"{prefix}   min={np.min(item):.4f}, max={np.max(item):.4f}")
                        
                        # Show all data elements
                        data = item[...]
                        if len(data.shape) == 1:
                            # 1D array - show all elements
                            print(f"{prefix}   ALL data: {data}")
                        elif len(data.shape) == 2:
                            # 2D array - show all rows
                            print(f"{prefix}   ALL rows:")
                            for i, row in enumerate(data):
                                print(f"{prefix}     [{i}]: {row}")
                        elif len(data.shape) == 3:
                            # 3D array - show all elements
                            print(f"{prefix}   ALL data:")
                            for i, elem in enumerate(data):
                                # if i ==20:
                                #     break
                                if name == 'state':
                                    print(f"{prefix}     [{i}]: {elem[:2]}")
                                else:
                                    print(f"{prefix}     [{i}]: {elem}")
                            # For goal keypoints specifically, check consistency
                            if 'goal' in name and 'keypoint' in name:
                                all_same = np.all([np.allclose(data[0], data[i]) for i in range(len(data))])
                                print(f"{prefix}   → All timesteps identical? {all_same}")
                        elif len(data.shape) == 4:
                            # 4D array (e.g., images) - show all element shapes (printing full data would be too large)
                            print(f"{prefix}   ALL element shapes:")
                            for i in range(len(data)):
                                print(f"{prefix}     [{i}] shape: {data[i].shape}")
                        else:
                            print(f"{prefix}   ALL data:")
                            for i, elem in enumerate(data):
                                print(f"{prefix}     [{i}]: {elem}")
    
    print(f"\n{'='*60}")
    print(f"Zarr file: {path}")
    print(f"{'='*60}\n")
    print_tree(root)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_zarr.py <path_to_zarr>")
        sys.exit(1)
    
    explore_zarr(sys.argv[1])