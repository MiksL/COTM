import torch
import threading
import queue
import time


class GPUInferenceServer:
    """GPU inference for processes batched requests"""
    def __init__(self, model, batch_size=1024, max_wait_ms=5):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.request_queue = queue.Queue() # Queue for incoming requests
        self.response_dict = {} # Dictionary for storing responses
        self.response_lock = threading.Lock()
        self.next_id = 0
        self.id_lock = threading.Lock()
        self.running = True
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def get_next_id(self):
        """Get a unique ID for a request"""
        with self.id_lock:
            id = self.next_id
            self.next_id += 1
        return id
    
    def submit(self, encoded_board):
        """Submit a board for inference and wait for result"""
        # Create a unique ID for this request
        request_id = self.get_next_id()
        
        # Create response event
        response_ready = threading.Event()
        
        # Register in response dictionary
        with self.response_lock:
            self.response_dict[request_id] = {
                'result': None,
                'event': response_ready
            }
        
        # Put request in queue
        self.request_queue.put((request_id, encoded_board))
        
        # Wait for response
        response_ready.wait()
        
        # Get result
        with self.response_lock:
            result = self.response_dict[request_id]['result']
            del self.response_dict[request_id]
        
        return result
    
    def _process_queue(self):
        """Worker thread that processes the queue"""
        while self.running:
            # Collect batch of requests
            batch_ids = []
            batch_boards = []
            start_time = None
            
            # Try to fill a batch or wait until max_wait_ms
            while len(batch_ids) < self.batch_size:
                try:
                    # Start timing after first item
                    if start_time is None and len(batch_ids) > 0:
                        start_time = time.time()
                    
                    # Timeout only applies after we have at least one item
                    timeout = None if start_time is None else max(0.001, self.max_wait_ms - (time.time() - start_time))
                    
                    # Try to get an item
                    request_id, encoded_board = self.request_queue.get(timeout=0.001 if start_time is None else timeout)
                    batch_ids.append(request_id)
                    batch_boards.append(encoded_board)
                    self.request_queue.task_done()
                    
                    # If we've waited long enough or batch is full, process it
                    if start_time is not None and ((time.time() - start_time) >= self.max_wait_ms or len(batch_ids) >= self.batch_size):
                        break
                
                except queue.Empty:
                    # If we have at least one item, process the batch
                    if len(batch_ids) > 0:
                        break
                    continue
            
            # Skip if no items collected
            if len(batch_ids) == 0:
                continue
            
            # Process batch on GPU
            try:
                # Stack boards into a batch
                boards_tensor = torch.stack([torch.tensor(board, dtype=torch.float32) for board in batch_boards])
                boards_tensor = boards_tensor.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    policies, values = self.model(boards_tensor)
                
                # Move results back to CPU
                policies = policies.cpu()
                values = values.cpu()
                
                # Distribute results
                with self.response_lock:
                    for i, request_id in enumerate(batch_ids):
                        if request_id in self.response_dict:
                            self.response_dict[request_id]['result'] = (policies[i], values[i])
                            self.response_dict[request_id]['event'].set()
            
            except Exception as e:
                # Handle errors in batch processing
                with self.response_lock:
                    for request_id in batch_ids:
                        if request_id in self.response_dict:
                            self.response_dict[request_id]['result'] = None
                            self.response_dict[request_id]['event'].set()
                print(f"Error in GPU inference: {e}")
    
    def shutdown(self):
        """Shut down the inference server"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)