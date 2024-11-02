cdef struct Bin:
    int start
    int end


def binning_2d(int[:, :] arr, int bins_per_dim):
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]

    cdef Bin[:, :] bins_x = <Bin[:, :] > malloc(bins_per_dim * sizeof(Bin))
    cdef Bin[:, :] bins_y = <Bin[:, :] > malloc(bins_per_dim * sizeof(Bin))

    cdef int bin_size_x = cols // bins_per_dim
    cdef int bin_size_y = rows // bins_per_dim

    cdef int i, j, x_bin_idx, y_bin_idx

    # Initialize bins
    for i in range(bins_per_dim):
        bins_x[i].start = i * bin_size_x
        bins_x[i].end = (i + 1) * bin_size_x

        bins_y[i].start = i * bin_size_y
        bins_y[i].end = (i + 1) * bin_size_y

    # Binning
    for i in range(rows):
        for j in range(cols):
            x_bin_idx = arr[i, j] // bin_size_x
            y_bin_idx = arr[i, j] // bin_size_y

            # Assign element to bin
            bins_x[x_bin_idx].elements.append(arr[i, j])
            bins_y[y_bin_idx].elements.append(arr[i, j])

    # Free memory and return bins
    bins = (bins_x, bins_y)
    free(bins_x)
    free(bins_y)

    return bins
