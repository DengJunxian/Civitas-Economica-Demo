#pragma once
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct Order {
  std::string order_id;
  std::string agent_id;
  double timestamp;
  std::string symbol;
  std::string side;
  std::string order_type;
  double price;
  double quantity; // Requested quantity (int in python usually, but double for
                   // safety)
  std::string status;
  double filled_qty;

  double remaining_qty() const { return quantity - filled_qty; }
  bool is_filled() const { return filled_qty >= quantity; }
};

struct Trade {
  std::string trade_id;
  double price;
  double quantity;
  std::string maker_id;
  std::string taker_id;
  std::string maker_agent_id;
  std::string taker_agent_id;
  double timestamp;
  double buyer_fee;
  double seller_fee;
  double seller_tax;
};

class LimitOrderBook {
public:
  LimitOrderBook(std::string symbol);
  ~LimitOrderBook() = default;

  // Core Actions
  std::vector<Trade> add_order(Order &order);
  bool cancel_order(std::string order_id);

  // Data Access
  double get_best_bid();
  double get_best_ask();
  std::map<std::string, std::vector<std::pair<double, double>>>
  get_depth(int levels);

  // Helpers
  void clear();

private:
  std::string symbol;

  // Helper types for order storage
  // Use shared_ptr to allow Orders to be in both map and id lookup, and updated
  // in place
  using OrderPtr = std::shared_ptr<Order>;

  // Bids: Descending price
  std::map<double, std::vector<OrderPtr>, std::greater<double>> bids;
  // Asks: Ascending price
  std::map<double, std::vector<OrderPtr>, std::less<double>> asks;

  // Lookup
  std::unordered_map<std::string, OrderPtr> order_lookup;

  // Matching logic
  std::vector<Trade> match_order(OrderPtr order);
  std::vector<Trade> match_against_asks(OrderPtr buy_order);
  std::vector<Trade> match_against_bids(OrderPtr sell_order);
  Trade execute_trade(OrderPtr buy_order, OrderPtr sell_order, OrderPtr taker,
                      OrderPtr maker);

  void add_to_book(OrderPtr order);
  void remove_order(OrderPtr order);
};
