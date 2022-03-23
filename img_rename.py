import os


def isImage(name):
    image_type = ['jpg', 'png', 'JPG', 'jpeg', ' PNG','bmp', 'tif']
    return True if name.split('.')[-1] in image_type else False

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self):
        self.path = '/mnt/bd/aurora-mtrc-data/datas/GIANA_challenge/segmentation/test_match/hdtest/masks'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if isImage(item):
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path), ''+str(i) + '.jpg')
                # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                #dst = os.path.join(os.path.abspath(self.path),  item.split('.')[0].zfill(5) + '.jpg')
                #dst = os.path.join(os.path.abspath(self.path),  item.split('.')[0] + '_t.tif')
                dst = os.path.join(os.path.abspath(self.path),  item.split('_')[0] + '.tif')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
