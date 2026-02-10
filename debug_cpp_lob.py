try:
    from core.exchange.order_book_cpp import OrderBookCPP
    lob = OrderBookCPP("TEST")
    res = lob._cpp_lob.get_depth(5)
    print(f"Type: {type(res)}")
    print(f"Value: {res}")
except ImportError:
    print("CPP module not found")
except Exception as e:
    print(f"Error: {e}")
