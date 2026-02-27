//
// Created by hzw on 2026/2/26.
//

#ifndef MY_INFERENCE_EXPR_H
#define MY_INFERENCE_EXPR_H
#include <memory>
#include <string>
#include <variant>

#include "expr/expr_op_type.h"

namespace my_inference {
    class Expr {
        class ExprImpl {
        public:
            explicit ExprImpl(const int64_t &value) : type(ExprOpType::Value), content_(value), cur_value_(value) {
            }

            explicit ExprImpl(const std::string &param) : type(ExprOpType::Param), content_(param) {
            }


            ExprImpl(const ExprOpType &type, const std::shared_ptr<ExprImpl> &l,
                     const std::shared_ptr<ExprImpl> &r) : type(type), content_(std::pair{l, r}) {
            }

            friend bool operator!=(const ExprImpl &e1, const ExprImpl &e2) {
                return e1.content_ != e2.content_;
            }

        private:
            using ContentType = std::variant<int64_t, std::string, std::pair<std::shared_ptr<ExprImpl>, std::shared_ptr<
                ExprImpl> > >;

            ExprOpType type;
            ContentType content_;
            int64_t cur_value_ = 0;
            friend class Expr;
        };

    public:
        explicit Expr(const int64_t &value) : impl_(std::make_shared<ExprImpl>(value)) {
        }

        explicit Expr(const std::string &param) : impl_(std::make_shared<ExprImpl>(param)) {
        }

        Expr(const ExprOpType &type, const Expr &l, const Expr &r) : impl_(
            std::make_shared<ExprImpl>(type, l.impl_, r.impl_)) {
        }

        [[nodiscard]] bool isValue() const {
            return impl_->type == ExprOpType::Value;
        }

        [[nodiscard]] bool isParam() const {
            return impl_->type == ExprOpType::Param;
        }

        [[nodiscard]] int64_t value() const {
            return std::get<int64_t>(impl_->content_);
        }

        [[nodiscard]] std::string param() const {
            return std::get<std::string>(impl_->content_);
        }

        friend bool operator!=(const Expr &e1, const Expr &e2) {
            return *(e1.impl_) != *(e2.impl_);
        }

        friend Expr operator+(const Expr &l, const Expr &r) {
            if (l.isValue() && r.isValue()) {
                return Expr(l.value() + r.value());
            }
            return {ExprOpType::Add, l, r};
        }

        friend Expr operator+(const Expr &l, const int64_t &r) {
            if (l.isValue()) {
                return Expr(l.value() + r);
            }
            return {ExprOpType::Add, l, Expr(r)};
        }

        friend Expr operator+(const int64_t &l, const Expr &r) {
            if (r.isValue()) {
                return Expr(l + r.value());
            }
            return {ExprOpType::Add, Expr(l), r};
        }

        friend Expr operator-(const Expr &l, const Expr &r) {
            if (l.isValue() && r.isValue()) {
                return Expr(l.value() - r.value());
            }
            return {ExprOpType::Sub, l, r};
        }

        friend Expr operator-(const Expr &l, const int64_t &r) {
            if (l.isValue()) {
                return Expr(l.value() - r);
            }
            return {ExprOpType::Sub, l, Expr(r)};
        }

        friend Expr operator-(const int64_t &l, const Expr &r) {
            if (r.isValue()) {
                return Expr(l - r.value());
            }
            return {ExprOpType::Sub, Expr(l), r};
        }

        friend Expr operator*(const Expr &l, const Expr &r) {
            if (l.isValue() && r.isValue()) {
                return Expr(l.value() * r.value());
            }
            return {ExprOpType::Mul, l, r};
        }

        friend Expr operator*(const Expr &l, const int64_t &r) {
            if (l.isValue()) {
                return Expr(l.value() * r);
            }
            return {ExprOpType::Mul, l, Expr(r)};
        }

        friend Expr operator*(const int64_t &l, const Expr &r) {
            if (r.isValue()) {
                return Expr(l * r.value());
            }
            return {ExprOpType::Mul, Expr(l), r};
        }

        friend Expr operator/(const Expr &l, const Expr &r) {
            if (l.isValue() && r.isValue()) {
                return Expr(l.value() * r.value());
            }
            return {ExprOpType::Div, l, r};
        }


        friend Expr operator/(const Expr &l, const int64_t &r) {
            if (l.isValue()) {
                return Expr(l.value() / r);
            }
            return {ExprOpType::Div, l, Expr(r)};
        }

        friend Expr operator/(const int64_t &l, const Expr &r) {
            if (r.isValue()) {
                return Expr(l / r.value());
            }
            return {ExprOpType::Div, Expr(l), r};
        }

    private:
        std::shared_ptr<ExprImpl> impl_;
    };
}
#endif //MY_INFERENCE_EXPR_H
