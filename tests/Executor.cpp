#include <catch2/catch.hpp>

#include <libkann/Variable.hpp>
#include <libkann/Function.hpp>

#include <libkann/operations/ReduceOperation.hpp>
#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/MatrixMultiplyOperation.hpp>

#include <libkann/Executor.hpp>

#include <fstream>

static constexpr size_t SIZE = 10;

TEST_CASE("NewAPI", "[NewAPI]")
{
  SECTION("Executor")
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
      std::ofstream file("output/executor1.dot");
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

  SECTION("Function")
  {
    class MyFunction : public kann::BinaryFunction
    {
    protected:
      std::shared_ptr<const kann::Variable> impl(std::shared_ptr<const kann::Variable> a, std::shared_ptr<const kann::Variable> b) const override
      {
        auto c = std::make_shared<const kann::Variable>(std::vector{a}, std::make_shared<kann::IdentityOperation>());
        auto d = std::make_shared<const kann::Variable>(std::vector{b}, std::make_shared<kann::IdentityOperation>());
        auto e = std::make_shared<const kann::Variable>(std::vector{a, b}, std::make_shared<kann::ReduceOperation>(2));

        auto f = std::make_shared<const kann::Variable>(std::vector{c, d}, std::make_shared<kann::ReduceOperation>(2));
        auto g = std::make_shared<const kann::Variable>(std::vector{d, e}, std::make_shared<kann::ReduceOperation>(2));

        auto h = std::make_shared<const kann::Variable>(std::vector{d, f, g}, std::make_shared<kann::ReduceOperation>(3));

        return h;
      }
    };

    const auto func = MyFunction();

    auto inputVariable1 = std::make_shared<const kann::Variable>();
    auto inputVariable2 = std::make_shared<const kann::Variable>();
    auto outputVariable = func.invoke({inputVariable1, inputVariable2});

    kann::Executor executor({inputVariable1, inputVariable2}, {outputVariable});
    {
      std::ofstream file("output/executor2.dot");
      executor.write_graphviz(file);
    }

    unsigned seed = GENERATE(take(100, random(0, INT_MAX)));
    srand(seed);

    kann::Tensor input1(SIZE), input2(SIZE);

    input1.asArray().setRandom();
    input2.asArray().setRandom();

    std::vector<kann::Tensor> inputs  = {input1, input2};
    std::vector<kann::Tensor> outputs = executor.evaluate(inputs);

    kann::Tensor output = outputs[0];

    REQUIRE(output.asVector().isApprox(input1.asVector() * 2.0 + input2.asVector() * 4.0));
  }

  SECTION("MatrixMultiply1")
  {
    auto op = std::make_shared<kann::MatrixMultiplyOperation>(10, 30, 20, false, false);

    auto a = std::make_shared<const kann::Variable>();
    auto b = std::make_shared<const kann::Variable>();
    auto c = std::make_shared<const kann::Variable>(std::vector{a,b}, op);

    kann::Executor executor({a,b}, {c});
    {
      std::ofstream file("output/executor3.dot");
      executor.write_graphviz(file);
    }

    unsigned seed = GENERATE(take(100, random(0, INT_MAX)));
    srand(seed);

    kann::Tensor input1(10*20), input2(20*30);

    input1.asArray().setRandom();
    input2.asArray().setRandom();

    std::vector<kann::Tensor> inputs  = {input1, input2};
    std::vector<kann::Tensor> outputs = executor.evaluate(inputs);

    kann::Tensor output = outputs[0];

    REQUIRE(output.asMatrix(10,30) == input1.asMatrix(10,20) * input2.asMatrix(20,30));
  }

  SECTION("MatrixMultiply2")
  {
    auto op = std::make_shared<kann::MatrixMultiplyOperation>(10, 30, 20, true, false);

    auto a = std::make_shared<const kann::Variable>();
    auto b = std::make_shared<const kann::Variable>();
    auto c = std::make_shared<const kann::Variable>(std::vector{a,b}, op);

    kann::Executor executor({a,b}, {c});
    {
      std::ofstream file("output/executor4.dot");
      executor.write_graphviz(file);
    }

    unsigned seed = GENERATE(take(100, random(0, INT_MAX)));
    srand(seed);

    kann::Tensor input1(10*20), input2(20*30);

    input1.asArray().setRandom();
    input2.asArray().setRandom();

    std::vector<kann::Tensor> inputs  = {input1, input2};
    std::vector<kann::Tensor> outputs = executor.evaluate(inputs);

    kann::Tensor output = outputs[0];

    REQUIRE(output.asMatrix(10,30) == input1.asMatrix(20,10).transpose() * input2.asMatrix(20,30));
  }

  SECTION("MatrixMultiply3")
  {
    auto op = std::make_shared<kann::MatrixMultiplyOperation>(10, 30, 20, false, true);

    auto a = std::make_shared<const kann::Variable>();
    auto b = std::make_shared<const kann::Variable>();
    auto c = std::make_shared<const kann::Variable>(std::vector{a,b}, op);

    kann::Executor executor({a,b}, {c});
    {
      std::ofstream file("output/executor5.dot");
      executor.write_graphviz(file);
    }

    unsigned seed = GENERATE(take(100, random(0, INT_MAX)));
    srand(seed);

    kann::Tensor input1(10*20), input2(20*30);

    input1.asArray().setRandom();
    input2.asArray().setRandom();

    std::vector<kann::Tensor> inputs  = {input1, input2};
    std::vector<kann::Tensor> outputs = executor.evaluate(inputs);

    kann::Tensor output = outputs[0];

    REQUIRE(output.asMatrix(10,30) == input1.asMatrix(10,20) * input2.asMatrix(30,20).transpose());
  }

  SECTION("MatrixMultiply4")
  {
    auto op = std::make_shared<kann::MatrixMultiplyOperation>(10, 30, 20, true, true);

    auto a = std::make_shared<const kann::Variable>();
    auto b = std::make_shared<const kann::Variable>();
    auto c = std::make_shared<const kann::Variable>(std::vector{a,b}, op);

    kann::Executor executor({a,b}, {c});
    {
      std::ofstream file("output/executor6.dot");
      executor.write_graphviz(file);
    }

    unsigned seed = GENERATE(take(100, random(0, INT_MAX)));
    srand(seed);

    kann::Tensor input1(10*20), input2(20*30);

    input1.asArray().setRandom();
    input2.asArray().setRandom();

    std::vector<kann::Tensor> inputs  = {input1, input2};
    std::vector<kann::Tensor> outputs = executor.evaluate(inputs);

    kann::Tensor output = outputs[0];

    REQUIRE(output.asMatrix(10,30) == input1.asMatrix(20,10).transpose() * input2.asMatrix(30,20).transpose());
  }
}
