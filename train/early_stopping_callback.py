class EarlyStoppingCallback:
    def __init__(self, patience: int, metric: str):
        assert patience > 0, "Patience must be greater than 0"
        self._patience = patience
        self._metric = metric
        self._max_value = float("-inf")

    def update(self, metrics: dict[str, float]):
        last_val = metrics[self._metric]
        if last_val < self._max_value:
            self._patience -=1
            if self._patience == 0:
                print(f"Stopping early due to no improvement in validation metric {self._metric} with top value recorded: {self._max_value}")
            return True

        self._max_value = last_val
        return False
