#计算均值
def average(img):
    (width, height) = img.shape
    avg = 0.0 
    for i in range(width):
        for j in range(height):
            avg += img[i][j]
    return avg/(width*height)

#计算方差
def variance(img, avg):
    (width, height) = img.shape
    var = 0.0
    for i in range(width):
        for j in range(height):
            var += (img[i][j]-avg)*(img[i][j]-avg)
    return (var/(width*height-1))**0.5

#计算协方差
def covariance(img1, avg1, img2, avg2):
    (width, height) = img1.shape
    cov = 0.0
    for i in range(width):
        for j in range(height):
            cov += (img1[i][j]-avg1)*(img2[i][j]-avg2)
    return cov/(width*height-1)

#计算ssim
def calculate(img1, img2):
    k1 = 0.01
    k2 = 0.03
    l = 255
    c1 = (k1 / l) * (k2 / l)
    c2 = (k2 / l) * (k2 / l)
    c3 = c2 / 2
    avg1 = average(img1)
    avg2 = average(img2)
    dev1 = variance(img1, avg1)
    dev2 = variance(img2, avg2)
    dev = covariance(img1, avg1, img2, avg2)
    #亮度
    luminance = (2*avg1*avg2+c1)/(avg1**2+avg2**2+c1)
    #对比度
    contrast = (2*dev1*dev2+c2)/(dev1**2+dev2**2+c2)
    #结构
    structure = (dev+c3)/(dev1*dev2+c3)
    ssim = luminance*contrast*structure
    return ssim
