//
// Created by hzw on 2026/2/23.
//
#pragma once
#include <string>
#include <utility>

#include "expr/expr.h"

namespace my_inference {
    class TensorDim {
    public:
        explicit TensorDim(int64_t value) : expr_(value) {
        }

        explicit TensorDim(const std::string &value) : expr_(value) {
        }

        explicit TensorDim(Expr expr) : expr_(std::move(expr)) {
        }

        [[nodiscard]] bool isDynamic() const {
            return !expr_.isValue();
        }

        [[nodiscard]] std::string param() const {
            return expr_.param();
        }

        [[nodiscard]] int64_t value() const {
            return expr_.value();
        }

        [[nodiscard]] bool isValue() const {
            return expr_.isValue();
        }

        [[nodiscard]] bool isClear() const {
            return !(expr_.isValue() && expr_.value() == 1);
        }

        TensorDim &operator=(const TensorDim &dim) = default;

        friend bool operator==(const TensorDim &td1, const TensorDim &td2) {
            return td1.expr_ == td2.expr_;
        }

        friend bool operator!=(const TensorDim &dim1, const TensorDim &dim2) {
            return dim1.expr_ != dim2.expr_;
        }

        friend TensorDim operator+(const TensorDim &l, const TensorDim &r) {
            return TensorDim(l.expr_ + r.expr_);
        }

        friend TensorDim operator+(const int64_t &l, const TensorDim &r) {
            return TensorDim(l + r.expr_);
        }

        friend TensorDim operator+(const TensorDim &l, const int64_t &r) {
            return TensorDim(l.expr_ + r);
        }

        friend TensorDim operator-(const TensorDim &l, const TensorDim &r) {
            return TensorDim(l.expr_ - r.expr_);
        }

        friend TensorDim operator-(const int64_t &l, const TensorDim &r) {
            return TensorDim(l - r.expr_);
        }

        friend TensorDim operator-(const TensorDim &l, const int64_t &r) {
            return TensorDim(l.expr_ - r);
        }

        friend TensorDim operator*(const TensorDim &l, const TensorDim &r) {
            return TensorDim(l.expr_ * r.expr_);
        }

        friend TensorDim operator*(const int64_t &l, const TensorDim &r) {
            return TensorDim(l * r.expr_);
        }

        friend TensorDim operator*(const TensorDim &l, const int64_t &r) {
            return TensorDim(l.expr_ * r);
        }

        friend TensorDim operator/(const TensorDim &l, const TensorDim &r) {
            return TensorDim(l.expr_ / r.expr_);
        }

        friend TensorDim operator/(const int64_t &l, const TensorDim &r) {
            return TensorDim(l / r.expr_);
        }

        friend TensorDim operator/(const TensorDim &l, const int64_t &r) {
            return TensorDim(l.expr_ / r);
        }

    private:
        Expr expr_;
    };
}
