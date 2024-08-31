from .data_structures import PriorityQueue, Node

class HuffmanEncoder:
  def __init__(self, data):
    nodes = [Node(data=x) for x in data]
    queue = PriorityQueue(nodes)
    self.root = self.generate_tree(queue)

    self.codes = self.depth_first_traverse()

  def generate_tree(self,queue):
    i = 0
    while len(queue) > 1:
      x_priority, x = queue.pull()
      y_priority, y = queue.pull()
      #print(x_priority,y_priority)

      # NOTE: I give the node a placeholder data string 
      # so the Python can distinguish nodes. Otherwise
      # it overwrites the nodes in insertion due to same 
      # hash of None.
      z = Node(data=f'{i}',left=x,right=y)
      x.set_parent(z)
      y.set_parent(z)
      queue.insert(z,x_priority + y_priority)
      i+=1

    root = queue.pull()[1]
    return root

  def depth_first_traverse(self):
    node = self.root
    codes = {}
    path = ''
    flag = True
    if isinstance(node.data, tuple):
      return {node.data:'0'}

    while flag:
      node.set_visited()
      left = node.get_left()
      right = node.get_right()


      if node.is_leaf():
        codes[node.get_data()] = path
        path = path[:-1] # Retreat
        node = node.get_parent()
      elif not right.is_visited():
        path += '1'
        node = node.get_right()
      elif not left.is_visited():
        path += '0'
        node = node.get_left()
      elif not node.is_root():
        path = path[:-1]
        node = node.get_parent()
      else:
        flag = False
    return codes

  def decode(self,code):
    node = self.root
    for i in range(len(code)):
      if node.is_leaf() or (code[i] not in ['0','1']):
        return 'Invalid code'
      if code[i] == '1':
        node = node.get_right()
      else:
        node = node.get_left()
    return node.get_data()

  def get_codes(self):
    return self.codes