def hist_distance(h1, h2, power_=2):
    #  print h1, h2
    needed, availiable = map(list, zip(*[(y - x if x >= y else 0, y - x if y >= x else 0) for x, y in zip(h1, h2)]))
    # print needed, availiable
    dist = 0
    j = 0
    for i in xrange(len(needed)):
        val = needed[i]
        while val != 0:
            if availiable[j] == 0:
                j += 1
                continue
            if availiable[j] >= abs(val):
                dist += abs(val) * abs(pow(i - j, power_))
                availiable[j] += val
                val = 0
            else:
                dist += abs(availiable[j]) * abs(pow(i - j, power_))
                # print availiable
                val += availiable[j]
                availiable[j] = 0

    return dist
