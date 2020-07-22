#pragma once

struct Selectable
{
  bool selected = false;

  void select()   { selected = true; }
  void deselect() { selected = false; }
};
