# Customizing Games

`IAlphazooGame` is the abstract game interface that AlphaZoo expects (it lists all the methods that must be implemented).

`PettingZooWrapper` is the standard implementation for PettingZoo AEC environments — it assumes "standard" PettingZoo behavior.

If your environment does not work exactly as `alphazoo` expects, you can override specific methods from the wrapper or implement `IAlphazooGame` from scratch.

```python
class MyPettingZooWrapper(PettingZooWrapper):
    def obs_to_state(self, obs, agent_id):
        # channels-last → channels-first
        t = torch.tensor(obs["observation"], dtype=torch.float32).unsqueeze(0)
        return t.permute(0, 3, 1, 2)
```

# Customizing networks