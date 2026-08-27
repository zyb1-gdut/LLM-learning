import matplotlib.pyplot as plt

if __name__ == '__main__':
    #读取现有的图片
    img_data = plt.imread("../data/img_1.jpg")
    print("图片的形状（HWC）",img_data.shape)

    #展示图片
    plt.imshow(img_data)
    plt.show()

    #保存图片
    plt.imsave("../data/img_1.jpg", img_data)

