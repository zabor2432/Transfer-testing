import tensorflow as tf

class Conv_2(tf.keras.Model):
    def __init__(self, num_classes=10, input_size=28):
        super(Conv_2, self).__init__()
        self.feat_size = 12544 if input_size == 32 else 12544 if input_size == 28 else -1
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, mask=None):
        x1 = self.conv1(x)
        if mask is not None:
            x1 = x1 * mask[0]
        x2 = tf.nn.relu(tf.nn.max_pool(x1, ksize=1, strides=1, padding='VALID'))
        x3 = self.conv2(x2)
        if mask is not None:
            x3 = x3 * mask[1]
        x4 = tf.nn.relu(tf.nn.max_pool(x3, ksize=2, strides=2, padding='VALID'))
        x4 = self.flatten(x4)
        x5 = self.fc1(x4)
        if mask is not None:
            x5 = x5 * mask[2]
        x6 = self.fc2(x5)
        if mask is not None:
            x6 = x6 * mask[3]
        x7 = self.fc3(x6)
        return x7

    def forward_features(self, x):
        x1 = tf.nn.relu(tf.nn.max_pool(self.conv1(x), ksize=1, strides=1, padding='VALID'))
        x2 = tf.nn.relu(tf.nn.max_pool(self.conv2(x1), ksize=2, strides=2, padding='VALID'))
        x2 = self.flatten(x2)
        x3 = self.fc1(x2)
        x4 = self.fc2(x3)
        x5 = self.fc3(x4)
        return [x2, x3, x4, x5]

    def forward_param_features(self, x):
        x1 = self.conv1(x)
        x2 = tf.nn.relu(tf.nn.max_pool(x1, ksize=1, strides=1, padding='VALID'))
        x3 = self.conv2(x2)
        x4 = tf.nn.relu(tf.nn.max_pool(x3, ksize=2, strides=2, padding='VALID'))
        x4 = self.flatten(x4)
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        x7 = self.fc3(x6)
        return [x1, x3, x5, x6, x7]