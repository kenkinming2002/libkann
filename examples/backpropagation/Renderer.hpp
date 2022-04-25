#pragma once

#include <SFML/Window/Event.hpp>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>

#include <functional>
#include <optional>
#include <thread>
#include <mutex>

class Renderer
{
public:
  static constexpr unsigned WINDOW_WIDTH = 800;
  static constexpr unsigned WINDOW_HEIGHT = 600;

public:
  Renderer()
    : m_thread(std::bind(&Renderer::run, this, std::placeholders::_1)) {}

public:
  struct Content
  {
    std::string title;
    std::vector<sf::Image> images;
  };

  void submit(Content content)
  {
    std::unique_lock lk(m_mutex);
    m_content = std::move(content);
  }

private:
  void run(std::stop_token token)
  {
    sf::RenderWindow window;
    window.create(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "AutoEncoder Visualization");

    std::vector<sf::Texture> textures;
    sf::Text text;

    sf::Font font;
    if(!font.loadFromFile("resources/fonts/NotoSansMono-Regular.ttf"))
      throw std::runtime_error("Failed to load font");

    text.setFont(font);

    size_t counter = 0;
    while(window.isOpen() && !token.stop_requested())
    {
      sf::Event evnt;
      while(window.pollEvent(evnt))
        if(evnt.type == sf::Event::Closed)
        {
          window.close();
          return;
        }

      window.clear();

      // Update textures
      if(counter++ == 1024)
      {
        counter = 0;

        std::unique_lock lk(m_mutex);
        if(m_content)
        {
          textures.resize(m_content->images.size());
          for(size_t i=0; i<m_content->images.size(); ++i)
            textures[i].loadFromImage(m_content->images[i]);

          text.setString(m_content->title);
        }
      }

      // Draw textures
      {
        window.setView(sf::View(sf::FloatRect(0.0f, 0.0f, 1.0f, 1.0f)));

        for(size_t i=0; i<textures.size(); ++i)
        {
          const float width = 1.0f / (float)textures.size(), height = 1.0f;
          const float x = width * i, y = 0.0f;

          sf::Sprite sprite;

          sprite.setTexture(textures[i]);
          sprite.setPosition(x, y);

          sf::FloatRect local_bounds = sprite.getLocalBounds();
          sprite.setScale(width / local_bounds.width, height / local_bounds.height);

          window.draw(sprite);
        }
      }

      // Draw text
      {
        sf::Vector2u windowSize = window.getSize();
        window.setView(sf::View(sf::FloatRect(0.0f, 0.0f, windowSize.x, windowSize.y)));
        window.draw(text);
      }

      window.display();
    }
  }

private:
  std::jthread m_thread;

private:
  std::mutex m_mutex;
  std::optional<Content> m_content;
};

