class Queue():
    def __init__(self):
        self.queue = []
        self.size = 0
        self.MAX_SIZE = 10
        self.front = 0
        
    def __len__(self):
        return self.size
    
    def isEmpty(self):
        return self.size == 0
    
    def isFull(self):
        return self.size >= self.MAX_SIZE
    
    def enqueue(self, data):
        if self.isFull():
            del self.queue[0]
            self.queue.append(data)
        else:
            self.queue.append(data)
            self.size += 1
    
    def peek(self):
        return self.queue[0]
    
    def equals(self):
        chk_data = self.peek()
        for i in self.queue[1:]:
            if chk_data != i:
                return False
        return True

    def get_list(self):
        return self.queue
    
q = Queue()
for i in range(100):
    q.enqueue(i)
    print(q.get_list(), len(q.get_list()))