from multiprocessing import Pool
import time
def plussum(num_list):
    time.sleep(1)
    result = num_list+10
    print(result)

if __name__ == '__main__':
    p = Pool(12)

    number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    time.time()
    p.map(plussum, number)
    p.close()
    p.join()

    print("finished")