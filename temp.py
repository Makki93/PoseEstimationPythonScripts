j = 0
for key in file_to_annot.keys():
    j += 1
    print(file_to_annot[key])

    if j == 20:
        quit()