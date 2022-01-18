#include <catch2/catch.hpp>

#include <libkann/Variable.hpp>
#include <libkann/Function.hpp>

#include <libkann/operations/ReduceOperation.hpp>
#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/MatrixMultiplyOperation.hpp>

#include <libkann/Differentiate.hpp>
#include <libkann/Executor.hpp>

#include <fstream>

static constexpr size_t SIZE = 10;

TEST_CASE("NewAPI", "[NewAPI]")
{
  SECTION("Executor")
  {
    auto a = std::make_shared<const kann::Variable>();
    auto b = std::make_shared<const kann::Variable>();

    auto c = std::make_shared<const kann::Variable>(std::vector{a}, std::make_shared<kann::IdentityOperation>(SIZE, SIZE, 0));
    auto d = std::make_shared<const kann::Variable>(std::vector{b}, std::make_shared<kann::IdentityOperation>(SIZE, SIZE, 0));
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

    auto input1 = std::make_shared<kann::Tensor>(SIZE);
    auto input2 = std::make_shared<kann::Tensor>(SIZE);

    input1->asArray().setRandom();
    input2->asArray().setRandom();

    auto inputs  = std::vector<std::shared_ptr<const kann::Tensor>>{input1, input2};
    auto outputs = executor.evaluate(inputs);

    auto output1 = outputs[0];
    auto output2 = outputs[1];

    REQUIRE(output1->asVector().isApprox(input1->asVector() + input2->asVector()));
    REQUIRE(output2->asVector().isApprox(input1->asVector() + input2->asVector() * 2.0));
  }

  SECTION("Gradients")
  {
    auto a = std::make_shared<const kann::Variable>();
    auto b = std::make_shared<const kann::Variable>();

    auto c = std::make_shared<const kann::Variable>(std::vector{a}, std::make_shared<kann::IdentityOperation>(SIZE, SIZE, 0));
    auto d = std::make_shared<const kann::Variable>(std::vector{b}, std::make_shared<kann::IdentityOperation>(SIZE, SIZE, 0));
    auto e = std::make_shared<const kann::Variable>(std::vector{a, b}, std::make_shared<kann::ReduceOperation>(2));

    auto f = std::make_shared<const kann::Variable>(std::vector{c, d}, std::make_shared<kann::ReduceOperation>(2));
    auto g = std::make_shared<const kann::Variable>(std::vector{d, e}, std::make_shared<kann::ReduceOperation>(2));

    auto h = std::make_shared<const kann::Variable>(std::vector{f, g}, std::make_shared<kann::MatrixMultiplyOperation>(10, 20, 30, false, false));
    auto hGradient = std::make_shared<const kann::Variable>();

    auto gradientsMap = kann::differentiate(std::vector{h}, std::vector{hGradient});

    std::vector<std::shared_ptr<const kann::Variable>> inputs;
    inputs.push_back(a);
    inputs.push_back(b);
    inputs.push_back(hGradient);

    std::vector<std::shared_ptr<const kann::Variable>> outputs;
    outputs.push_back(gradientsMap.at(a));
    outputs.push_back(gradientsMap.at(b));
    outputs.push_back(h);

    kann::Executor executor(inputs, outputs);
    {
      std::ofstream file("output/executor_gradients.dot");
      executor.write_graphviz(file);
    }
  }

  SECTION("Function")
  {
    class MyFunction : public kann::BinaryFunction
    {
    protected:
      std::shared_ptr<const kann::Variable> impl(std::shared_ptr<const kann::Variable> a, std::shared_ptr<const kann::Variable> b) const override
      {
        auto c = std::make_shared<const kann::Variable>(std::vector{a}, std::make_shared<kann::IdentityOperation>(SIZE,SIZE,0));
        auto d = std::make_shared<const kann::Variable>(std::vector{b}, std::make_shared<kann::IdentityOperation>(SIZE,SIZE,0));
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

    auto input1 = std::make_shared<kann::Tensor>(SIZE);
    auto input2 = std::make_shared<kann::Tensor>(SIZE);

    input1->asArray().setRandom();
    input2->asArray().setRandom();

    auto inputs  = std::vector<std::shared_ptr<const kann::Tensor>>{input1, input2};
    auto outputs = executor.evaluate(inputs);

    auto output = outputs[0];

    REQUIRE(output->asVector().isApprox(input1->asVector() * 2.0 + input2->asVector() * 4.0));
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

    auto input1 = std::make_shared<kann::Tensor>(10 * 20);
    auto input2 = std::make_shared<kann::Tensor>(20 * 30);

    input1->asArray().setRandom();
    input2->asArray().setRandom();

    auto inputs  = std::vector<std::shared_ptr<const kann::Tensor>>{input1, input2};
    auto outputs = executor.evaluate(inputs);

    auto output = outputs[0];

    REQUIRE(output->asMatrix(10,30) == input1->asMatrix(10,20) * input2->asMatrix(20,30));
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

    auto input1 = std::make_shared<kann::Tensor>(10 * 20);
    auto input2 = std::make_shared<kann::Tensor>(20 * 30);

    input1->asArray().setRandom();
    input2->asArray().setRandom();

    auto inputs  = std::vector<std::shared_ptr<const kann::Tensor>>{input1, input2};
    auto outputs = executor.evaluate(inputs);

    auto output = outputs[0];

    REQUIRE(output->asMatrix(10,30) == input1->asMatrix(20,10).transpose() * input2->asMatrix(20,30));
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

    auto input1 = std::make_shared<kann::Tensor>(10 * 20);
    auto input2 = std::make_shared<kann::Tensor>(20 * 30);

    input1->asArray().setRandom();
    input2->asArray().setRandom();

    auto inputs  = std::vector<std::shared_ptr<const kann::Tensor>>{input1, input2};
    auto outputs = executor.evaluate(inputs);

    auto output = outputs[0];

    REQUIRE(output->asMatrix(10,30) == input1->asMatrix(10,20) * input2->asMatrix(30,20).transpose());
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

    auto input1 = std::make_shared<kann::Tensor>(10 * 20);
    auto input2 = std::make_shared<kann::Tensor>(20 * 30);

    input1->asArray().setRandom();
    input2->asArray().setRandom();

    auto inputs  = std::vector<std::shared_ptr<const kann::Tensor>>{input1, input2};
    auto outputs = executor.evaluate(inputs);

    auto output = outputs[0];

    REQUIRE(output->asMatrix(10,30) == input1->asMatrix(20,10).transpose() * input2->asMatrix(30,20).transpose());
  }
}
