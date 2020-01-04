# dense-comparison-experiments
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
