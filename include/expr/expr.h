//
// Created by hzw on 2026/2/26.
//

#ifndef MY_INFERENCE_EXPR_H
#define MY_INFERENCE_EXPR_H
#include <memory>
#include <string>
#include <variant>

#include "expr/expr_type.h"

namespace my_inference {
    class Expr {
        class ExprImpl {
        public:
            explicit ExprImpl(const int64_t &value) : type(ExprType::Value), content_(value), cur_value_(value) {
            }

            explicit ExprImpl(const std::string &param) : type(ExprType::Param), content_(param) {
            }


            ExprImpl(const ExprType &type, const std::shared_ptr<ExprImpl> &l,
                     const std::shared_ptr<ExprImpl> &r) : type(type), content_(std::pair{l, r}) {
            }

            friend bool operator==(const ExprImpl &e1, const ExprImpl &e2) {
                return e1.content_ == e2.content_;
            }

            friend bool operator!=(const ExprImpl &e1, const ExprImpl &e2) {
                return e1.content_ != e2.content_;
            }

        private:
            using ContentType = std::variant<int64_t, std::string, std::pair<std::shared_ptr<ExprImpl>, std::shared_ptr<
                ExprImpl> > >;

            ExprType type;
            ContentType content_;
            int64_t cur_value_ = 0;
            friend class Expr;
        };

    public:
        explicit Expr(const int64_t &value) : impl_(std::make_shared<ExprImpl>(value)) {
        }

        explicit Expr(const std::string &param) : impl_(std::make_shared<ExprImpl>(param)) {
        }

        Expr(const ExprType &type, const Expr &l, const Expr &r) : impl_(
            std::make_shared<ExprImpl>(type, l.impl_, r.impl_)) {
        }

        [[nodiscard]] bool isValue() const {
            return impl_->type == ExprType::Value;
        }

        [[nodiscard]] bool isParam() const {
            return impl_->type == ExprType::Param;
        }

        [[nodiscard]] int64_t value() const {
            return std::get<int64_t>(impl_->content_);
        }

        [[nodiscard]] std::string param() const {
            return std::get<std::string>(impl_->content_);
        }

        friend bool operator==(const Expr &e1, const Expr &e2) {
            return *e1.impl_ == *e2.impl_;
        }

        friend bool operator!=(const Expr &e1, const Expr &e2) {
            return *e1.impl_ != *e2.impl_;
        }

        friend Expr operator+(const Expr &l, const Expr &r) {
            return make<ExprType::Add>(l, r);
        }

        friend Expr operator+(const Expr &l, const int64_t &r) {
            return make<ExprType::Add>(l, r);
        }

        friend Expr operator+(const int64_t &l, const Expr &r) {
            return make<ExprType::Add>(l, r);
        }

        friend Expr operator-(const Expr &l, const Expr &r) {
            return make<ExprType::Sub>(l, r);
        }

        friend Expr operator-(const Expr &l, const int64_t &r) {
            return make<ExprType::Sub>(l, r);
        }

        friend Expr operator-(const int64_t &l, const Expr &r) {
            return make<ExprType::Sub>(l, r);
        }

        friend Expr operator*(const Expr &l, const Expr &r) {
            return make<ExprType::Mul>(l, r);
        }

        friend Expr operator*(const Expr &l, const int64_t &r) {
            return make<ExprType::Mul>(l, r);
        }

        friend Expr operator*(const int64_t &l, const Expr &r) {
            return make<ExprType::Mul>(l, r);
        }

        friend Expr operator/(const Expr &l, const Expr &r) {
            return make<ExprType::Div>(l, r);
        }


        friend Expr operator/(const Expr &l, const int64_t &r) {
            return make<ExprType::Div>(l, r);
        }

        friend Expr operator/(const int64_t &l, const Expr &r) {
            return make<ExprType::Div>(l, r);
        }

    private:
        template<ExprType Type, typename T, typename U>
        static Expr make(const T &l, const U &r) {
            if (isValue(l) && isValue(r)) {
                if constexpr (Type == ExprType::Add) { return Expr(asValue(l) + asValue(r)); }
                if constexpr (Type == ExprType::Sub) { return Expr(asValue(l) - asValue(r)); }
                if constexpr (Type == ExprType::Mul) { return Expr(asValue(l) * asValue(r)); }
                if constexpr (Type == ExprType::Div) { return Expr(asValue(l) / asValue(r)); }
            }
            return Expr(Type, asExpr(l), asExpr(r));
        }

        static bool isValue(int64_t) {
            return true;
        }

        static bool isValue(const Expr &expr) {
            return expr.isValue();
        }

        static Expr asExpr(const int64_t &value) {
            return Expr(value);
        }


        static const Expr &asExpr(const Expr &expr) {
            return expr;
        }

        static int64_t asValue(const int64_t &value) {
            return value;
        }


        static int64_t asValue(const Expr &expr) {
            return expr.value();
        }


        std::shared_ptr<ExprImpl> impl_;
    };
}
#endif //MY_INFERENCE_EXPR_H
