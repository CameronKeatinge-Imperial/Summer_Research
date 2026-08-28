import heapq
import numpy as np
from src2.queue_object import PriorityQueue

class MultiPriorityQueue():
    '''
    Plan - pass in the dictionary, then can lookup and get the node'''
    def __init__(self,target_distributions = "None"):
        if target_distributions == "None":
            self.hyperedge_queue = PriorityQueue()
            self.cardinality_segmentation = False
        else:
            self.hyperedge_queues = {i: PriorityQueue() for i, value in enumerate(target_distributions, start=0) if value != 0}
            self.cardinality_segmentation = True

    def push(self, score, node, dict=None):
        if self.cardinality_segmentation == False:
            self.hyperedge_queue.push_func(score, node)
        else:
            #get cardinality from the node itself
            cardinality = self.cardinality_of_hyperedge(node,dict)
            self.hyperedge_queues[cardinality].push_func(score, node)

    def push_mult(self, pairs,node_to_hyperedge_dictionary=None):
        for score, node in pairs:
            self.push(score, node, node_to_hyperedge_dictionary)

    def peek_lowest_score(self,i,cardinality=None):
        if self.cardinality_segmentation == False:
            return self.hyperedge_queue.peek_lowest_score(i)
        else:
            return self.hyperedge_queues[cardinality].peek_lowest_score(i)

    def extract_lowest_score(self,cardinality=None):
        if self.cardinality_segmentation == False:
            return self.hyperedge_queue.extract_lowest_score()
        else:
            return self.hyperedge_queues[cardinality].extract_lowest_score()

    def is_empty(self):
        if self.cardinality_segmentation == False:
            return self.hyperedge_queue.is_empty()
        else:
            #if all keys return zero
            #return self.hyperedge_queues[cardinality].is_empty()
            return all(q.is_empty() for q in self.hyperedge_queues.values())

    def cardinality_of_hyperedge(self,node,node_dictonary):
        hyperedge_of_interest = node_dictonary[node]
        return len(hyperedge_of_interest.split(",")) if isinstance(hyperedge_of_interest, str) else len(hyperedge_of_interest)
    
    def is_empty_cardinality(self,cardinality):
        return self.hyperedge_queues[cardinality].is_empty()

    def peek_lowest_score(self,cardinality=None):
        if cardinality==None:
            return self.hyperedge_queue.peek_lowest_score()
        else:
            return self.hyperedge_queues[cardinality].peek_lowest_score()
