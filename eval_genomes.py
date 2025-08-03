import slimevolleygym
import neat
import os

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # 각 유전체에 대해 게임 환경 생성
        env = slimevolleygym.SlimeVolleyEnv()

        # NEAT 신경망 생성
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # 게임 시작 및 첫 상태 관찰
        observation = env.reset()

        cumulative_reward = 0.0
        genome.fitness = 0.0 # 피트니스 초기화

        while True:
            # env.render() # 학습 과정을 보고 싶을 때 주석 해제

            # 현재 관찰 값을 신경망에 입력하여 행동 결정
            output = net.activate(observation)

            # 출력을 [0, 1, 0] 과 같은 이진 행동으로 변환
            action = [1 if o > 0.5 else 0 for o in output]

            # 결정한 행동을 환경에 실행
            observation, reward, done, info = env.step(action)

            cumulative_reward += reward

            # 게임이 끝나면 루프 종료
            if done:
                break

        # 누적 보상을 해당 유전체의 피트니스로 할당
        genome.fitness = cumulative_reward
        env.close()

# --- NEAT 실행 코드는 이전과 동일 ---
def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # 300 세대 동안 학습 실행
    winner = p.run(eval_genomes, 300)

    # 결과 출력
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    # 바로 이 부분!
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
