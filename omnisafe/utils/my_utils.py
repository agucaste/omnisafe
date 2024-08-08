def unwrap_env(env):
    """
    Unwraps an environment, so as to
    Args:
        env (MyOffPolicyAdapter): The environment

    Returns:
        The unwrapped environment
    """
    while hasattr(env, '_env'):
        env = env._env
    return env
