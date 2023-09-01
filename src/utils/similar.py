def compute_distance_behavior_states(y1, y2):
    """
    Calculate the coverage score between two lists, y1 and y2.

    Args:
        y1 (list): The first list.
        y2 (list): The second list.

    Returns:
        float: The coverage score between y1 and y2.
    """
    # Get the lengths of y1 and y2
    y1_length = len(y1)
    y2_length = len(y2)

    # Calculate the coverage score as the absolute difference between the lengths
    coverage_score = abs(y1_length - y2_length)

    # Determine the common length between y1 and y2
    common_length = min(y1_length, y2_length)

    # Slice y1 and y2 up to the common length
    y1_common = y1[:common_length]
    y2_common = y2[:common_length]

    # Iterate over the common length and compare each element of y1_common and y2_common
    for i in range(common_length):
        y1_e = y1_common[i]
        y2_e = y2_common[i]

        # If the elements are equal, continue to the next iteration
        if y1_e == y2_e:
            continue
        else:
            # If the elements are different, increment the coverage score by 1
            coverage_score += 1

    # Normalize the coverage score by dividing it by the maximum length of y1 and y2
    coverage_score /= float(max(y1_length, y2_length))

    # Format the coverage score to a float with 4 decimal places
    score = float(format(coverage_score, '.4f'))

    return score


if __name__ == '__main__':
    a = [[1, 2, 3], [4, 5, 6]]
    b = [[1, 2, 3], [4, 5, 6]]
    c = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    e = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    s = compute_distance_behavior_states(a, e)
    print(s)