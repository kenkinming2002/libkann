#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Window/Event.hpp>

// These are merely suggestion to the window manager and need not be obeyed
static constexpr unsigned WINDOW_WIDTH = 800;
static constexpr unsigned WINDOW_HEIGHT = 600;

int main()
{
  sf::RenderWindow window;
  window.create(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "AutoEncoder Visualization");

  while(window.isOpen())
  {
    sf::Event evnt;
    while(window.pollEvent(evnt))
      switch(evnt.type)
      {
      case sf::Event::Closed:
        window.close();
        break;
      default:
        break;
      }

    window.clear();
    window.display();
  }
}
