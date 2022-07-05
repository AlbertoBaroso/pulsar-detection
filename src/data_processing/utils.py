# Reshape a one-dimensional array to a column array
def vcol(array):
    return array.reshape((array.size, 1))


# Reshape a column array to a row array
def vrow(array):
    return array.reshape((1, array.size))


def one_dimensional_array(array):
    return array.reshape((array.size,))