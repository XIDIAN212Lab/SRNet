import torch
import numpy as np

def box2gaussian(boxes1,boxes2):
    """将两个边界框转化为旋转高斯分布
    boxes:[-1,5],math: x,y,w,h,theta
    return:math:mu, sigama
    """
    x1, y1, w1, h1, theta1 = torch.unbind(boxes1,dim=1)
    x2, y2, w2, h2, theta2 = torch.unbind(boxes2, dim=1)
    x1 = torch.reshape(x1, [-1, 1])
    y1 = torch.reshape(y1, [-1, 1])
    h1 = torch.reshape(h1, [-1, 1])
    w1 = torch.reshape(w1, [-1, 1])
    theta1 = torch.reshape(theta1, [-1, 1])
    x2 = torch.reshape(x2, [-1, 1])
    y2 = torch.reshape(y2, [-1, 1])
    h2 = torch.reshape(h2, [-1, 1])
    w2 = torch.reshape(w2, [-1, 1])
    theta2 = torch.reshape(theta2, [-1, 1])

    sigma1_1 = w1 / 2 * torch.cos(theta1) ** 2 + h1 / 2 * torch.sin(theta1) ** 2
    sigma1_2 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_3 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_4 = w1 / 2 * torch.sin(theta1) ** 2 + h1 / 2 * torch.cos(theta1) ** 2
    sigma1 = torch.reshape(torch.cat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], dim=-1), [-1, 2, 2])

    sigma2_1 = w2 / 2 * torch.cos(theta2) ** 2 + h2 / 2 * torch.sin(theta2) ** 2
    sigma2_2 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_3 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_4 = w2 / 2 * torch.sin(theta2) ** 2 + h2 / 2 * torch.cos(theta2) ** 2
    sigma2 = torch.reshape(torch.cat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], dim=-1), [-1, 2, 2])


    return x1, y1, x2, y2, sigma1, sigma2

def qbox2gaussian(boxes, num_pts=4):
    """

    boxes:math: x1,y1,x2,y2,x3,y3,x4,y4   shape:[-1, 8]
    """
    x = torch.reshape(torch.mean(boxes[:, ::2].float(), dim=1), [-1, 1])
    y = torch.reshape(torch.mean(boxes[:, 1::2].float(), dim=1), [-1, 1])
    mu = torch.reshape(torch.cat([x, y], dim=-1), [-1, 1, 2])
    a = torch.reshape(boxes[:, :2], [-1,1, 2]) - mu
    a = torch.reshape(a,[-1, 2, 1])
    sigma = torch.matmul(a, torch.reshape(boxes[:, :2], [-1, 1, 2]) - mu)
    for n in range(num_pts - 1):
        e = torch.reshape(boxes[:, (n + 1) * 2:(n + 2) * 2], [-1, 1, 2]) - mu
        # e = torch.reshape(e,[-1, 2, 1])
        f = torch.reshape(e,[-1,2, 1])
        sigma += torch.matmul(e,
                                 f)
    sigma /= num_pts

    return mu, sigma

def kullback_leibler_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
    """
    Calculate the kullback-leibler divergence between two Gaussian distributions : :math:`\mathbf D_{kl} = 0.5*((\mathbf \mu_{1}-\mathbf \mu_{2})^T \mathbf \Sigma_{2}^{1/2}(\mathbf \mu_{1}-\mathbf \mu_{2})+0.5*\mathbf Tr(\mathbf \Sigma_{2}^{-1} \mathbf \Sigma_{1})+0.5*\ln |\mathbf \Sigma_{2}|/|\mathbf \Sigma_{1}| -1`

    :param mu1: mean :math:`(\mu_{1})` of the Gaussian distribution, shape: [-1, 1, 2]
    :param mu2: mean :math:`(\mu_{2})` of the Gaussian distribution, shape: [-1, 1, 2]
    :param mu1_T: transposition of :math:`(\mu_{1})`, shape: [-1, 2, 1]
    :param mu2_T: transposition of :math:`(\mu_{2})`, shape: [-1, 2, 1]
    :param sigma1: covariance :math:`(\Sigma_{1})` of the Gaussian distribution, shape: [-1, 2, 2]
    :param sigma2: covariance :math:`(\Sigma_{1})` of the Gaussian distribution, shape: [-1, 2, 2]
    :return:  kullback-leibler divergence, :math:`\mathbf D_{kl}`
    """

    sigma1_square = torch.matmul(sigma1, sigma1)
    sigma2_square = torch.matmul(sigma2, sigma2)
    a = torch.matmul(torch.inverse(sigma2_square), sigma1_square)
    n = a.shape[0]
    item1 = torch.tensor([n,1])
    for i in range(n):
        item1[i] = torch.trace(a[i])

    item2 = torch.matmul(torch.matmul(mu2 - mu1, torch.inverse(sigma2_square)), mu2_T - mu1_T)
    item3 = torch.log(torch.det(sigma2_square) / torch.det(sigma1_square))
    item1 = torch.reshape(item1, [-1, ])
    item2 = torch.reshape(item2, [-1, ])
    item3 = torch.reshape(item3, [-1, ])
    return (item1 + item2 + item3 - 2) / 2.

def gaussian_kullback_leibler_divergence(boxes1, boxes2):
    """
    Calculate the kullback-leibler divergence between boxes1 and boxes2

    :param boxes1: :math:`(x_{1},y_{1},w_{1},h_{1},\theta_{1})`, shape: [-1, 5]
    :param boxes2: :math:`(x_{2},y_{2},w_{2},h_{2},\theta_{2})`, shape: [-1, 5]
    :return: kullback-leibler divergence, :math:`\mathbf D_{kl}`
    """

    x1, y1, x2, y2, sigma1, sigma2 = box2gaussian(boxes1, boxes2)

    mu1 = torch.reshape(torch.cat([x1, y1], dim=-1), [-1, 1, 2])
    mu2 = torch.reshape(torch.cat([x2, y2], dim=-1), [-1, 1, 2])

    mu1_T = torch.reshape(torch.cat([x1, y1], dim=-1), [-1, 2, 1])
    mu2_T = torch.reshape(torch.cat([x2, y2], dim=-1), [-1, 2, 1])

    kl_divergence = torch.reshape(kullback_leibler_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
    return kl_divergence


if __name__ == '__main__':
    # rbox = np.array([[0, 0, 40, 20, -1 * np.pi / 180]])
    # x1, y1, x2, y2, sigma1, sigma2 = box2gaussian(torch.from_numpy(rbox), torch.from_numpy(rbox))
    # print(sigma1)
    # print("___________________")
    # qbox = np.array([[-10, 20, -10, -20, 10, -20, 10, 20]])
    # num_pts = 4
    # print(qbox2gaussian(torch.from_numpy(qbox), num_pts))

    rbox1 = np.array([[0, 0, 40, 20, -1 * np.pi / 180], [0, 0, 40, 20, -1 * np.pi / 180]])
    rbox2 = np.array([[0, 0, 40, 20, -1 * np.pi / 180], [0, 0, 40, 20, -1 * np.pi / 180]])

    loss = gaussian_kullback_leibler_divergence(torch.from_numpy(rbox1),torch.from_numpy(rbox2))
    print("____________________________________")

    print("loss", loss)