import tensorflow as tf

class DispNet:
    def __init__(self):
        self.img1 = tf.placeholder(tf.float32, shape=[None, 768,384,3], name="img1")
        self.img2 = tf.placeholder(tf.float32, shape=[None, 768,384,3], name="img2")
        self.disp = tf.placeholder(tf.float32, shape=[None, 768,384,1], name="disp")
        self.weights = tf.placeholder(tf.float32, shape=[6], name="weights")
        imgs = tf.concat([self.img1, self.img2], axis=3)

        conv1 = self.conv_relu(imgs, 7, 2, 64, "conv1")
        conv2 = self.conv_relu(conv1, 5, 2, 128, "conv2")
        conv3a = self.conv_relu(conv2, 5, 2, 256, "conv3a")
        conv3b = self.conv_relu(conv3a, 3, 1, 256, "conv3b")
        conv4a = self.conv_relu(conv3b, 3, 2, 512, "conv4a")
        conv4b = self.conv_relu(conv4a, 3, 1, 512, "conv4b")
        conv5a = self.conv_relu(conv4b, 3, 2, 512, "conv5a")
        conv5b = self.conv_relu(conv5a, 3, 1, 512, "conv5b")
        conv6a = self.conv_relu(conv5b, 3, 2, 1024, "conv6a")
        conv6b = self.conv_relu(conv6a, 3, 1, 1024, "conv6b")
        self.pr6 = self.conv(conv6b, 3, 1, 1, name="pr6")
        self.loss6 = self.l1loss(self.pr6, self.disp, "loss6")

        upconv5 = self.upconv_relu(conv6b, 4, 2, 512, "upconv5")
        pr6_up = self.upconv(self.pr6, 4, 2, 1, name="pr6_up")
        iconv5 = self.upconv_relu(self.concat([upconv5, pr6_up, conv5b], axis=3), 3, 1, 512, "iconv5")
        self.pr5 = self.conv(iconv5, 3, 1, 1, name="pr5")
        self.loss5 = self.l1loss(self.pr5, self.disp, "loss5")

        upconv4 = self.upconv_relu(iconv5, 4, 2, 256, "upconv4")
        pr5_up = self.upconv(self.pr5, 4, 2, 1, name="pr5_up")
        iconv4 = self.upconv_relu(self.concat([upconv4, pr5_up, conv4b], axis=3), 3, 1, 256, "iconv4")
        self.pr4 = self.conv(iconv4, 3, 1, 1, name="pr4")
        self.loss4 = self.l1loss(self.pr4, self.disp, "loss4")

        upconv3 = self.upconv_relu(iconv4, 4, 2, 128, "upconv3")
        pr4_up = self.upconv(self.pr4, 4, 2, 1, name="pr4_up")
        iconv3 = self.upconv_relu(self.concat([upconv3, pr4_up, conv3b], axis=3), 3, 1, 128, "iconv3")
        self.pr3 = self.conv(iconv3, 3, 1, 1, name="pr3")
        self.loss3 = self.l1loss(self.pr3, self.disp, "loss3")

        upconv2 = self.upconv_relu(iconv3, 4, 2, 64, "upconv2")
        pr3_up = self.upconv(self.pr3, 4, 2, 1, name="pr3_up")
        iconv2 = self.upconv_relu(self.concat([upconv2, pr3_up, conv2], axis=3), 3, 1, 64, "iconv2")
        self.pr2 = self.conv(iconv2, 3, 1, 1, name="pr2")
        self.loss2 = self.l1loss(self.pr2, self.disp, "loss2")

        upconv1 = self.upconv_relu(iconv2, 4, 2, 32, "upconv1")
        pr2_up = self.upconv(self.pr2, 4, 2, 1, name="pr2_up")
        iconv1 = self.upconv_relu(self.concat([upconv1, pr2_up, conv1], axis=3), 3, 1, 32, "iconv1")
        self.pr1 = self.conv(iconv1, 3, 1, 1, name="pr1")
        self.loss1 = self.l1loss(self.pr1, self.disp, "loss1")

        self.loss = self.loss6 * self.weights[0]

        for w, l in zip(range(1,6),[self.loss5, self.loss4, self.loss3, self.loss2, self.loss1]):
            self.loss += l * self.weights[w]

        self.add_summaries()

    def add_summaries(self):
        tf.summary.scalar("loss1", self.loss1)
        tf.summary.scalar("loss2", self.loss2)
        tf.summary.scalar("loss3", self.loss3)
        tf.summary.scalar("loss4", self.loss4)
        tf.summary.scalar("loss5", self.loss5)
        tf.summary.scalar("loss6", self.loss6)

        tf.summary.image("pr1", self.pr1)
        tf.summary.image("pr2", self.pr2)
        tf.summary.image("pr3", self.pr3)
        tf.summary.image("pr4", self.pr4)
        tf.summary.image("pr5", self.pr5)
        tf.summary.image("pr6", self.pr6)

    def conv(self, inputs, kernel_size, stride, channels, activation=None, name=None):
        return tf.layers.conv2d(inputs=inputs,
                                filters=channels,
                                kernel_size=(kernel_size, kernel_size),
                                strides=(stride, stride),
                                padding='same',
                                activation=activation,
                                name=name)


    def conv_relu(self, inputs, kernel_size, stride, channels, name):
        return self.conv(inputs, kernel_size, stride, channels, activation=tf.nn.relu, name=name)

    def upconv(self, input, kernel_size, stride, channels, activation=None, name=None):
        return tf.layers.conv2d_transpose(inputs=input,
                                          filters=channels,
                                          kernel_size=(kernel_size, kernel_size),
                                          strides=(stride, stride),
                                          padding='same',
                                          activation=activation,
                                          name=name)

    def upconv_relu(self, inputs, kernel_size, stride, channels, name=None):
        return self.upconv(inputs, kernel_size, stride, channels, activation=tf.nn.relu, name=name)

    def l1loss(self, pred, gt, name):
        pred_shape = pred.get_shape()
        gt = tf.image.resize_images(gt, pred_shape[1:3])
        return tf.reduce_sum(tf.abs(pred - gt), name=name)

    def concat(self, values, axis):
        return tf.concat(values, axis)