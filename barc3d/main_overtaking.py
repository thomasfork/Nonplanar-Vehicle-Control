'''
Script for demonstrating raceline computation and use for overtaking
'''
from barc3d.surfaces import load_surface
from barc3d.games.overtaking import OvertakingConfig, StackelbergOvertaking

def main():
    '''
    Examples of nonplanar overtaking
    '''
    surface = 'l_track'   # collision avoidance without overtaking example
    surface = 'tube_turn' # overtaking example
    
    config = OvertakingConfig()
    config.ego_config.v0   = 10
    config.target_config.v0   = 10
    if surface == 'l_track':
        config.sf = 90
        config.ego_config.y0   = 0
        config.ego_config.ths0 = 0
        config.target_config.y0   = 0
        config.target_config.ths0 = 0
    elif surface == 'tube_turn':
        config.sf = 200
        config.ego_config.s0   = 0
        config.ego_config.y0   = -8
        config.ego_config.ths0 = 0
        config.target_config.s0   = 10
        config.target_config.y0   = -8
        config.target_config.ths0 = 0

        config.ego_config.v0   = 40
        config.target_config.v0   = 40
    else:
        pass
    
    # uncomment to make target and ego use the same model
    #config.ego_config.agent_class = config.target_config.agent_class
    
     
    surf = load_surface(surface)

    overtaking = StackelbergOvertaking(surf, config)
    overtaking.solve()

if __name__ == '__main__':
    main()
    
