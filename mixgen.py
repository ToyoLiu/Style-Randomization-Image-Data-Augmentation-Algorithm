import random
import numpy as np


def mixgen(image, text, num, lam=0.5):
    """
    This function performs data augmentation by mixing images and concatenating text labels within a dataset.
    The mixup is performed only on a subset of the dataset defined by `num`.

    Args:
    image (numpy array): The array containing image data.
    text (list of str): The list containing text data.
    num (int): Number of initial elements in the dataset to mix with the next `num` elements.
    lam (float, optional): The lambda parameter defines the mixing ratio. Defaults to 0.5.

    Returns:
    tuple: A tuple containing the mixed images and concatenated text.
    """
    # Loop through the first 'num' elements to perform mixing
    for i in range(num):
        # Mix images using a weighted average defined by 'lam'
        image[i, :] = lam * image[i, :] + (1 - lam) * image[i + num, :]
        # Concatenate the text of the current index with the text of the element at index 'i+num'
        text[i] = text[i] + " " + text[i + num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    """
    This function performs data augmentation by randomly mixing images and concatenating text labels across the entire batch.

    Args:
    image (tensor): A tensor of image data with shape [batch_size, other_dimensions].
    text (list of str): The list of text labels for the batch.
    num (int): Unused parameter, present for interface consistency.
    lam (float, optional): The lambda parameter defines the mixing ratio. Defaults to 0.5.

    Returns:
    tuple: A tuple containing the mixed images and concatenated text.
    """
    # Determine the batch size from the first dimension of the image tensor
    batch_size = image.size()[0]
    # Generate a random permutation of indices from 0 to batch_size - 1
    index = np.random.permutation(batch_size)
    # Loop through each element in the batch to perform random mixup
    for i in range(batch_size):
        # Mix images using a weighted average and a randomly selected index
        image[i, :] = lam * image[i, :] + (1 - lam) * image[index[i], :]
        # Concatenate the text of the current index with the text of a randomly selected index
        text[i] = text[i] + " " + text[index[i]]
    return image, text

