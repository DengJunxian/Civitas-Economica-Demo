#include "lob.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_civitas_lob, m) {
  m.doc() = "Civitas C++ Limit Order Book using pybind11";

  py::class_<Order>(m, "Order")
      .def(py::init<>())
      .def_readwrite("order_id", &Order::order_id)
      .def_readwrite("agent_id", &Order::agent_id)
      .def_readwrite("timestamp", &Order::timestamp)
      .def_readwrite("symbol", &Order::symbol)
      .def_readwrite("side", &Order::side)
      .def_readwrite("order_type", &Order::order_type)
      .def_readwrite("price", &Order::price)
      .def_readwrite("quantity", &Order::quantity)
      .def_readwrite("status", &Order::status)
      .def_readwrite("filled_qty", &Order::filled_qty)
      .def_property_readonly("remaining_qty", &Order::remaining_qty)
      .def_property_readonly("is_filled", &Order::is_filled);

  py::class_<Trade>(m, "Trade")
      .def(py::init<>())
      .def_readwrite("trade_id", &Trade::trade_id)
      .def_readwrite("price", &Trade::price)
      .def_readwrite("quantity", &Trade::quantity)
      .def_readwrite("maker_id", &Trade::maker_id)
      .def_readwrite("taker_id", &Trade::taker_id)
      .def_readwrite("maker_agent_id", &Trade::maker_agent_id)
      .def_readwrite("taker_agent_id", &Trade::taker_agent_id)
      .def_readwrite("timestamp", &Trade::timestamp)
      .def_readwrite("buyer_fee", &Trade::buyer_fee)
      .def_readwrite("seller_fee", &Trade::seller_fee)
      .def_readwrite("seller_tax", &Trade::seller_tax);

  py::class_<LimitOrderBook>(m, "LimitOrderBook")
      .def(py::init<std::string>())
      .def("add_order", &LimitOrderBook::add_order)
      .def("cancel_order", &LimitOrderBook::cancel_order)
      .def("get_best_bid", &LimitOrderBook::get_best_bid)
      .def("get_best_ask", &LimitOrderBook::get_best_ask)
      .def("get_depth", &LimitOrderBook::get_depth)
      .def("clear", &LimitOrderBook::clear)
      .def("add_order_exploded",
           [](LimitOrderBook &self, std::string order_id, std::string agent_id,
              double timestamp, std::string symbol, std::string side,
              std::string order_type, double price, double quantity) {
             Order o;
             o.order_id = order_id;
             o.agent_id = agent_id;
             o.timestamp = timestamp;
             o.symbol = symbol;
             o.side = side;
             o.order_type = order_type;
             o.price = price;
             o.quantity = quantity;
             o.filled_qty = 0;
             o.status = "pending";

             // Call core
             std::vector<Trade> trades = self.add_order(o);

             // Return tuple of (filled_qty, status, trade_list_of_tuples)
             py::list py_trades;
             for (auto &t : trades) {
               py_trades.append(py::make_tuple(
                   t.trade_id, t.price, t.quantity, t.maker_id, t.taker_id,
                   t.maker_agent_id, t.taker_agent_id, t.timestamp, t.buyer_fee,
                   t.seller_fee, t.seller_tax));
             }

             return py::make_tuple(o.filled_qty, o.status, py_trades);
           });
}
