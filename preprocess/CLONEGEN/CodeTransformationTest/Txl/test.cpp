// 测试用例
ptr1.reset(new Foo(1, 2));
ptr2.reset( new Bar<std::string>(arg) );
ptr3.reset(new (std::nothrow) Baz());
