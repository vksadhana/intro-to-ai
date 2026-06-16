class CustomPQ_maxG:

    def __init__(self):
        self.heap = []

    def push(self, f_val, g_val, cell):
        new_item = (f_val, -g_val, cell)
        self.heap.append(new_item)

        i = len(self.heap) - 1
        while i > 0:
            parent_index = (i - 1) // 2
            if self.heap[i] < self.heap[parent_index]:
                temp = self.heap[i]
                self.heap[i] = self.heap[parent_index]
                self.heap[parent_index] = temp
                i = parent_index
            else:
                break

    def pop(self):
        if len(self.heap) == 1:
            return self.heap.pop()

        top_item = self.heap[0]
        last_item = self.heap.pop()
        self.heap[0] = last_item

        i = 0
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < len(self.heap):
                if self.heap[left] < self.heap[smallest]:
                    smallest = left

            if right < len(self.heap):
                if self.heap[right] < self.heap[smallest]:
                    smallest = right

            if smallest == i:
                break

            temp = self.heap[i]
            self.heap[i] = self.heap[smallest]
            self.heap[smallest] = temp
            i = smallest

        return top_item

    def is_empty(self):
        if len(self.heap) == 0:
            return True
        return False

    def __len__(self):
        return len(self.heap)


class CustomPQ_minG:

    def __init__(self):
        self.heap = []

    def push(self, f_val, g_val, cell):
        new_item = (f_val, g_val, cell)
        self.heap.append(new_item)

        i = len(self.heap) - 1
        while i > 0:
            parent_index = (i - 1) // 2
            if self.heap[i] < self.heap[parent_index]:
                temp = self.heap[i]
                self.heap[i] = self.heap[parent_index]
                self.heap[parent_index] = temp
                i = parent_index
            else:
                break

    def pop(self):
        if len(self.heap) == 1:
            return self.heap.pop()

        top_item = self.heap[0]
        last_item = self.heap.pop()
        self.heap[0] = last_item

        i = 0
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < len(self.heap):
                if self.heap[left] < self.heap[smallest]:
                    smallest = left

            if right < len(self.heap):
                if self.heap[right] < self.heap[smallest]:
                    smallest = right

            if smallest == i:
                break

            temp = self.heap[i]
            self.heap[i] = self.heap[smallest]
            self.heap[smallest] = temp
            i = smallest

        return top_item

    def is_empty(self):
        if len(self.heap) == 0:
            return True
        return False

    def __len__(self):
        return len(self.heap)