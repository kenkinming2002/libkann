#include "App.hpp"

#include "Config.hpp"

#include <SFML/Window/Event.hpp>

App::App(seed_type seed)
  : m_world(World::Config{
      .width = CONFIG.world.width,
      .height = CONFIG.world.height,
      .initialCreaturesCount = CONFIG.world.initialCreaturesCount,
      .initialBerryBushesClusterSizeMin = CONFIG.world.initialBerryBushesClusterSizeMin,
      .initialBerryBushesClusterSizeMax = CONFIG.world.initialBerryBushesClusterSizeMax,
      .initialBerryBushesClusterCount   = CONFIG.world.initialBerryBushesClusterCount
    }, Creature::Config{
      .maxRadius                     = CONFIG.creature.radius, // TODO: Fix the name
      .maxEnergy                     = CONFIG.creature.maxEnergy,
      .maxHealth                     = CONFIG.creature.maxHealth,
      .forwardLinearSpeed            = CONFIG.creature.forwardLinearSpeed,
      .backwardLinearSpeed           = CONFIG.creature.backwardLinearSpeed,
      .angularSpeed                  = CONFIG.creature.angularSpeed,
      .passiveEnergyDrain            = CONFIG.creature.passiveEnergyDrain,
      .movementEnergyDrainMultiplier = CONFIG.creature.movementEnergyDrainMultiplier,
      .healingThreshold              = CONFIG.creature.healingThreshold,
      .healingRate                   = CONFIG.creature.healingRate,
      .hungerHealthDrain             = CONFIG.creature.hungerHealthDrain,
      .eatingCooldown                = CONFIG.creature.eatingCooldown,
      .matingCooldown                = CONFIG.creature.matingCooldown,
      .viewDistance                  = CONFIG.creature.viewDistance,
    }, Creature::ModelConfig{
      .hiddenLayers = CONFIG.creature.hiddenLayers,
      .memory       = CONFIG.creature.memory
    }, BerryBush::Config{
      .energyPerBerry = CONFIG.berryBush.energyPerBerry,
      .maxBerryCount  = CONFIG.berryBush.maxBerryCount,
      .growthRate     = CONFIG.berryBush.growthRate,
      .radius         = CONFIG.berryBush.radius
    }),
    m_csvFile("statistics.csv"),
    m_window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "App"),
    m_renderer(m_window) {}

void App::run()
{
  m_world.beginStatistics(m_csvFile);
  while(this->loop());
}

bool App::loop()
{
  handleInput();
  update();
  render();
  return m_window.isOpen();
}

void App::handleInput()
{
  sf::Event event;
  while (m_window.pollEvent(event))
  {
    if(m_renderer.handleInput(event))
      continue;

    switch(event.type)
    {
      case sf::Event::KeyPressed:
        switch(event.key.code)
        {
          case sf::Keyboard::Space:
            this->toggleSpeed();
            break;
          default:
            break;
        }
        break;
      case sf::Event::Closed:
        m_window.close();
        break;
      default:
        break;
    }
  }
}

void App::update()
{
  switch(m_speedMode)
  {
  case SpeedMode::NORMAL:
    m_elapsedtime += m_clock.restart().asSeconds();
    while(m_elapsedtime>=FIXED_DELTA_TIME)
    {
      m_elapsedtime-=FIXED_DELTA_TIME;
      m_world.update(FIXED_DELTA_TIME);
      m_world.writeStatistics(m_csvFile);
    }
    break;
  case SpeedMode::ASAP:
    while(m_clock.getElapsedTime().asSeconds() <= FRAME_TIME)
    {
      m_world.update(FIXED_DELTA_TIME);
      m_world.writeStatistics(m_csvFile);
    }

    m_clock.restart();
    break;
  }
}

void App::render() const
{
  m_renderer.begin();
  m_world.draw(m_renderer);
  m_renderer.end();

  m_window.display();
}

void App::toggleSpeed()
{
  m_speedMode = m_speedMode == SpeedMode::NORMAL ? SpeedMode::ASAP : SpeedMode::NORMAL;
}
