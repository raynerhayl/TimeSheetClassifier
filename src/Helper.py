def print_matrix(m):
    print("Length: {0}".format(len(m)))

    for val in m:
        line = ""
        for token in val:
            line += token
        print(line)

def list_difference(l1, l2):
    """""
        Returns all elements in l1 not in l2
    """
    difference = []

    for i in range(0, len(l1)):
            exists = False
            print(i)
            for row in l2:
                row_equivalent = True
                for col in range(0, len(row)):
                    if not row[col] == l1[i][col]:
                        row_equivalent = False
                if row_equivalent:
                    exists = True
            if not exists:
                difference.append(l1[i])

    return difference
