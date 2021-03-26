# dense-comparison-experiments

Use dense for fully connected layers please!

```
  num_hid = 200

  # input layer
  self.state_input = tf.placeholder("float",[None,self.state_dim])

  if not CONFIG["dense"]:
      self.W1 = self.weight_variable([self.state_dim,num_hid])
      self.b1 = self.bias_variable([num_hid])
      self.W2 = self.weight_variable([num_hid,num_hid])
      self.b2 = self.bias_variable([num_hid])
      self.W3 = self.weight_variable([num_hid,self.action_dim])
      self.b3= self.bias_variable([self.action_dim])

      # hidden layers
      h_layer = tf.nn.relu(tf.matmul(self.state_input,self.W1) + self.b1)
      h_layer02 = tf.nn.relu(tf.matmul(h_layer,self.W2) + self.b2)

      # Q Value layer
      self.Q_value = tf.matmul(h_layer02,self.W3) + self.b3

  else:
      h_layer = tf.layers.dense( self.state_input, num_hid, activation=tf.nn.relu )
      h_layer2 = tf.layers.dense( h_layer, num_hid, activation=tf.nn.relu )
      self.Q_value = tf.layers.dense( h_layer2, self.action_dim )
```

# Analysis

## The kernel_initializer used in tf.layers.dense

kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.compat.v1.get_variable`.
        
        
If initializer is `None` (the default), the default initializer passed in
    the variable scope will be used. If that one is `None` too, a
    `glorot_uniform_initializer` will be used.

# TO DO
Initialize the weights using glorot_uniform_initializer, and to see whether the performance can be well as the one using tf.layers.dense.

