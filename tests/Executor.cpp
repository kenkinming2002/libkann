#include <catch2/catch.hpp>

#include <libkann/Variable.hpp>
#include <libkann/operations/ReduceOperation.hpp>
#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/Executor.hpp>

#include <fstream>

static constexpr size_t SIZE = 10;

TEST_CASE("Executor", "[Executor]")
{
  auto a = std::make_shared<const kann::Variable>();
  auto b = std::make_shared<const kann::Variable>();

  auto c = std::make_shared<const kann::Variable>(std::vector{a}, std::make_shared<kann::IdentityOperation>());
  auto d = std::make_shared<const kann::Variable>(std::vector{b}, std::make_shared<kann::IdentityOperation>());
  auto e = std::make_shared<const kann::Variable>(std::vector{a, b}, std::make_shared<kann::ReduceOperation>(2));

  auto f = std::make_shared<const kann::Variable>(std::vector{c, d}, std::make_shared<kann::ReduceOperation>(2));
  auto g = std::make_shared<const kann::Variable>(std::vector{d, e}, std::make_shared<kann::ReduceOperation>(2));

  kann::Executor executor({a, b}, {f, g});
  {
    std::ofstream file("output/executor.dot");
    executor.write_graphviz(file);
  }

  unsigned seed = GENERATE(take(100, random(0, INT_MAX)));
  srand(seed);

  kann::Tensor input1(SIZE), input2(SIZE);

  input1.asArray().setRandom();
  input2.asArray().setRandom();

  std::vector<kann::Tensor> inputs  = {input1, input2};
  std::vector<kann::Tensor> outputs = executor.evaluate(inputs);

  kann::Tensor output1 = outputs[0], output2 = outputs[1];

  REQUIRE(output1.asVector().isApprox(input1.asVector() + input2.asVector()));
  REQUIRE(output2.asVector().isApprox(input1.asVector() + input2.asVector() * 2.0));
}
