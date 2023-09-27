import csv
import random
import glob

# for multishot reid
# 提取定量的正负样本并存储为文件

ROOT = 'C:\\data\\MARS'
output = 'C:\\data\\MARS\\recs'


def load_split():
    train, test = [], []
    cnt = 0
    for i in range(386):
        cam_a = glob.glob('%s/multi_shot/cam_a/person_%04d/*.png' % (ROOT, i))
        cam_b = glob.glob('%s/multi_shot/cam_b/person_%04d/*.png' % (ROOT, i))
        if len(cam_a) * len(cam_b) > 0:
            cnt += 1
            if cnt > 100:
                test.append(i)
            else:
                train.append(i)
            if cnt >= 200:
                break
    return train, test


def rnd_pos(N, i):
    x = random.randint(0, N - 2)
    return x + 1 if x == i else x


def save_rec(lst, path, name):
    lst_file = '%s/%s.lst' % (path, name)
    rec_file = '%s/%s.rec' % (path, name)
    # print(lst_file, rec_file, '%s %s %s %s resize=128 quality=90' % (im2rec, lst_file, ROOT, rec_file))
    fo = csv.writer(open(lst_file, "w"), delimiter='\t', lineterminator='\n')
    for item in lst:
        fo.writerow(item)
    # print('echo 123456 | sudo -S %s %s %s %s resize=128 quality=90 &' % (im2rec, lst_file, ROOT, rec_file))
    # subprocess.call('%s %s %s %s resize=128 quality=90' % (im2rec, lst_file, ROOT, rec_file))


def save_train(f, is_valid=False):
    plst, nlst, cnt, N, pool = [], [], 0, len(f), [_ for _ in range(len(f))]
    for _ in range(100000 if not is_valid else 2000):
        ts = random.sample(pool, 96)  # 随机返回96个不同pid信息
        ns, ps = ts[:64], ts[64:]  # ns：64个， ps：32个
        for r in range(32):
            i, x, y = ps[r], ns[r + r], ns[r + r + 1]  # 3个随机pid
            p1c = random.randint(0, len(f[i]) - 1)
            p2c = rnd_pos(len(f[i]), p1c)  # 在f[i]的pid下对应的camera中任选2个
            p1 = (cnt, i, f[i][p1c][random.randint(0, len(f[i][p1c]) - 1)])  # 在pid==i&camera=random1==>任选一张照片
            p2 = (cnt + 1, i, f[i][p2c][random.randint(0, len(f[i][p2c]) - 1)])
            n1c = random.randint(0, len(f[x]) - 1)
            n2c = random.randint(0, len(f[y]) - 1)
            n1 = (cnt, x, f[x][n1c][random.randint(0, len(f[x][n1c]) - 1)])
            n2 = (cnt + 1, y, f[y][n2c][random.randint(0, len(f[y][n2c]) - 1)])
            cnt += 2
            plst.append(p1)
            plst.append(p2)
            nlst.append(n1)
            nlst.append(n2)  # plst顺序读取时每两张图像都是同一id，nlst顺序读取时每两张图像都是不同id
    # plst与nlst是等长的，is_valid=False时，长度为6,400,000，is_valid=True时，长度为128,000
    save_rec(plst, output, 'image_' + ('valid' if is_valid else 'train') + '_even')
    save_rec(nlst, output, 'image_' + ('valid' if is_valid else 'train') + '_rand')


def gen_train():
    pool = []
    for i in range(1500):  # 因为 train 中的 625个文件，文件名在：0001-1499 之间
        images = glob.glob('%s\\bbox_train\\%04d\\*.jpg' % (ROOT, i))
        f = dict()  # 根据摄像头id给图像归类
        for k in images:  # images：同一行人id的所有图像
            name = k.split('\\')[-1]
            ct = name[4:6]  # camera id
            if not ct in f:
                f[ct] = []
            f[ct].append(k[len(ROOT):])  # /bbox_train/%04d/*.jpg
        g = []  # 去除掉只拍摄当前行人单张图像的摄像头，存储到g
        for x in f:
            if len(f[x]) > 1:
                g.append(f[x])
        if len(g) <= 1:
            continue
        pool.append(g)  # pool 中存储所有行人的信息，每个元素代表一个字典，存储某个行人对应摄像头下的所有图像（只有单张图像的则去除）

    save_train(pool)  # len(pool) == 619
    save_train(pool, is_valid=True)


def naive_lst(dataset):
    lst_file = open('%s/MARS-evaluation/info/%s_name.txt' % (ROOT, dataset))
    lst, cnt = [], 0
    for line in lst_file:
        s = line.strip()
        lst.append((cnt, 0, '/bbox_%s/%s/%s' % (dataset, s[:4], s)))
        cnt += 1
    save_rec(lst, output, 'eval_' + dataset)


if __name__ == '__main__':
    # naive_lst('train')
    # naive_lst('test')
    gen_train()