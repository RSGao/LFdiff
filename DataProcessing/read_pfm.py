import os 
import numpy as np

def _get_next_line(f):
    next_line = f.readline().rstrip()
    # ignore comments
    while next_line.startswith('#'.encode()):
        next_line = f.readline().rstrip()
    return next_line.decode()

def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = os.path.join(r'D:\additional_depth_disp_all_views\additional_depth_disp_all_views\antinous\gt_depth_lowres_Cam001.pfm')
    depth = read_pfm(path)
    plt.imshow(depth)
    plt.show()
    print(depth)
    print(depth.shape)