class ActionAbstraction:
    ABSTRACTIONS = {
        'Kuhn': ['check', 'bet']  # First decision point
    }
    
    def __init__(self):
        """Kuhn Poker simplified actions"""
        self.name = 'Kuhn'
        self.actions = self.ABSTRACTIONS['Kuhn']
    
    def translate_action(self, abstract_action, env):
        """No translation needed - direct 1:1 mapping"""
        return abstract_action