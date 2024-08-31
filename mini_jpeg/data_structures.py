from collections import Counter

class PriorityQueue:
    # NOTE: Our queue will always work on fixed size arrays
    # of at most 64 elements coming from 8x8 block matrices.
    # Hence running time isn't bottlenecked by queue and we
    # implement it in array and not by heap for log complexity.
    def __init__(self, data):
      self.queue = dict(Counter(data))

    def insert(self, node, frequency):
      self.queue[node] = frequency
      #print(self.queue)

    def pull(self):
      key = min(self.queue,key=self.queue.get)
      value = self.queue[key]
      del self.queue[key]
      return (value,key)

    def __len__(self):
      return len(self.queue)

class Node:
  def __init__(self, data, parent=None, left=None, right=None, visited=False):
    self.data = data
    self.parent = parent
    self.left = left
    self.right = right
    self.visited = visited

  def is_root(self):
    return self.parent == None

  def is_leaf(self):
    return (self.left == None) and (self.right == None)

  def set_visited(self):
    self.visited = True

  def is_visited(self):
    return self.visited

  def get_data(self):
    return self.data

  def set_parent(self, node):
    self.parent = node

  def set_left(self, node):
    self.left = node

  def set_right(self, node):
    self.right = node

  def get_left(self):
    return self.left

  def get_right(self):
    return self.right

  def get_parent(self):
    return self.parent

  def __hash__(self):
    return hash(self.data)

  def __eq__(self, x):
    if isinstance(x, Node):
      return self.data == x.data
    return False

  def __repr__(self):
    return f"Node(data={self.data!r})"