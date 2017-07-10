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

def write_matrix(m, file_name):
    file = open(file_name, 'w+')
    for row in range(0, m.shape[0]):
        for col in range(0, m.shape[1]):
            file.write(str(m[row,col]))
            if col < m.shape[1] - 1:
                file.write(', ')
            else:
                file.write('\n')